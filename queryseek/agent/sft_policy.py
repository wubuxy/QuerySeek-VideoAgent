"""Training-only annotation teacher and a shared, label-free deployment prompt."""
from __future__ import annotations

from functools import lru_cache
import json

from queryseek.agent.anchored_policy import ANCHOR_PROMPT, AnchoredPolicy, support_cells
from queryseek.agent.search_policy import (
    SearchState, allowed_actions, node_window, select_observations, transition)
from queryseek.eval.qvhighlights import temporal_iou

SFT_PROMPT = ANCHOR_PROMPT.replace(
    "Abstain if evidence is insufficient.",
    "If evidence is insufficient and useful search actions remain, continue searching. "
    "Abstain only when no useful search remains within the budget.")


class SearchPolicy(AnchoredPolicy):
    """The same prompt for the quantized base and every trained adapter."""
    def __init__(self, backend, frames, duration):
        def adapt(prompt, observed, schema):
            if not prompt.startswith(ANCHOR_PROMPT):
                raise ValueError("Unexpected anchor prompt")
            return backend(SFT_PROMPT+prompt[len(ANCHOR_PROMPT):], observed, schema)
        super().__init__(adapt, frames, duration)


class AnnotationTeacher:
    """Offline weak targets only. Never instantiate this in model evaluation."""
    def __init__(self, frames, duration, gt_windows, cfg, method):
        self.frames, self.duration, self.cfg, self.method = frames, duration, cfg, method
        self.by_id = {f.frame_id:f for f in frames}
        self.gt = gt_windows
        if not gt_windows or any(not 0 <= a < b <= duration for a,b in gt_windows):
            raise ValueError("Invalid training labels")
        self.cells = support_cells(frames,duration)
        self.examples = []

    @lru_cache(maxsize=4096)
    def best_answer(self, observed_ids):
        frames = sorted((self.by_id[i] for i in observed_ids),key=lambda f:(f.timestamp_sec,f.frame_id))
        best = (0.0,None)
        for i,first in enumerate(frames):
            for j in range(i,len(frames)):
                last = frames[j]
                for a,b in self.gt:
                    # Both cited endpoints must lie inside the same annotated occurrence.
                    if a <= first.timestamp_sec < b and a <= last.timestamp_sec < b:
                        score = temporal_iou(self.cells[first.frame_id][0],self.cells[last.frame_id][1],a,b)
                        if score > best[0]:
                            best = (score,f"answer:{i}:{j}")
        return best

    def _state(self,path,ids):
        return SearchState("offline teacher",self.duration,self.method,path,
                           [self.by_id[i] for i in ids])

    @lru_cache(maxsize=8192)
    def plan(self,path,ids,turn):
        state = self._state(path,ids)
        quality,answer = self.best_answer(tuple(sorted(ids)))
        actions = allowed_actions(state,self.frames,turn,self.cfg)
        if quality >= .7:
            return quality,0,answer
        candidates = [(quality,0,answer)] if quality >= .5 else [(0.,0,"abstain")]
        for action in actions:
            if action in {"answer","abstain"}:
                continue
            nxt = self._state(path,ids)
            transition(nxt,action)
            fresh = select_observations(self.frames,nxt,min(self.cfg["frames_per_turn"],
                                        self.cfg["max_frames"]-len(ids)))
            if not fresh:
                continue
            next_ids = tuple(ids)+tuple(f.frame_id for f in fresh)
            q,steps,_ = self.plan(nxt.path,next_ids,turn+1)
            candidates.append((q,steps+1,action))
        # Reward correct localization, penalize unnecessary tool use.
        chosen = max(candidates,key=lambda x:(x[0]-.025*x[1],-x[1]))
        if chosen[0] < .5 and len(actions) > 2:
            # Do not teach omniscient early rejection of an unobserved event.
            navigation = []
            for action in actions:
                if action in {"answer","abstain"}:
                    continue
                nxt = self._state(path,ids);transition(nxt,action)
                fresh = select_observations(self.frames,nxt,min(self.cfg["frames_per_turn"],
                                            self.cfg["max_frames"]-len(ids)))
                if not fresh:
                    continue
                positives = sum(any(a<=f.timestamp_sec<b for a,b in self.gt) for f in fresh)
                w = node_window(nxt.path,self.duration)
                overlap = sum(max(0,min(w[1],b)-max(w[0],a)) for a,b in self.gt)/(w[1]-w[0])
                navigation.append((positives,overlap,action))
            if navigation:
                return 0.,1,max(navigation,key=lambda x:(x[0],x[1]))[2]
        return chosen

    def __call__(self,prompt,observed,schema):
        payload = json.loads(prompt[len(SFT_PROMPT):])
        ids = tuple(f.frame_id for f in observed)
        turn = self.cfg["max_turns"]-payload["remaining_calls_including_this"]
        _,_,choice = self.plan(tuple(payload["tree_path"]),ids,turn)
        if choice not in schema["properties"]["choice"]["enum"]:
            raise ValueError("Teacher action not in deployment schema")
        self.examples.append({"prompt":prompt,"frame_ids":list(ids),"target":{"choice":choice},
                              "turn":turn,"path":payload["tree_path"],
                              "teacher":"offline_annotation_weak_supervision"})
        return {"choice":choice}
