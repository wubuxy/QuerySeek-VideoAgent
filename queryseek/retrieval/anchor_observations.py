"""Preserve an observed seed when allocating a fixed candidate observation budget."""
from copy import deepcopy
from queryseek.retrieval.evidence_packets import packet_frames_valid


def retain_seed(packet,seed_frame,seed_sha256,duration):
    packet_frames_valid(packet,duration)
    t=seed_frame['timestamp_sec'];a,b=packet['interval_sec']
    if (packet['origin']!='seed_window' or packet.get('seed_time_sec') is None
            or abs(t-packet['seed_time_sec'])>1e-6 or not a<=t<b or not seed_sha256):
        raise ValueError('Seed must match the actually observed triggering frame')
    result=deepcopy(packet)
    index=min(range(len(result['frames'])),key=lambda i:(abs(result['frames'][i]['timestamp_sec']-t),
                                                        result['frames'][i]['timestamp_sec']))
    previous_time=result['frames'][index]['timestamp_sec']
    result['frames'][index]={**seed_frame,'frame_id':packet['candidate_id']+'_seed'}
    result['frame_sha256'][index]=seed_sha256
    pairs=sorted(zip(result['frames'],result['frame_sha256']),key=lambda p:p[0]['timestamp_sec'])
    result['frames']=[f for f,h in pairs];result['frame_sha256']=[h for f,h in pairs]
    result['sampling_revision']='anchor-preserved-v1'
    packet_frames_valid(result,duration)
    return result,{'seed_source_time_sec':t,'seed_image_sha256':seed_sha256,
                   'replaced_source_time_sec':previous_time,'frame_budget_changed':False}
