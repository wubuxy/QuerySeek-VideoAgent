import copy
import io
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
from scripts import univtg_input_transfer as study
from queryseek.ingest.cache import write_json_atomic
from queryseek.ingest.ffprobe import sha256_file
from queryseek.retrieval.univtg_inputs import upstream_temporal_inputs


def zip_prefix(value):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as file:
        file.writestr('archive/data.pkl',pickle.dumps(value,protocol=4))
    return stream.getvalue()


def test_checkpoint_metadata_never_executes_pickle_and_does_not_guess_epoch(tmp_path):
    marker=tmp_path/'executed'
    class Payload:
        def __reduce__(self):
            return (Path.touch,(marker,))
    report=study.prefix_epoch(zip_prefix({'model':Payload(),'epoch':7}))
    assert report['epoch_candidate']==7 and not marker.exists()
    assert study.prefix_epoch(zip_prefix({'n_epoch':10}))['epoch_candidate'] is None
    assert study.prefix_epoch(b'<html>confirmation</html>')['epoch_candidate'] is None
    assert study.prefix_epoch(zip_prefix({'epoch':8})[:40])['epoch_candidate'] is None


def test_torch_streaming_zip_unsized_header_can_be_statically_inspected():
    blob=bytearray(zip_prefix({'epoch':9}))
    struct.pack_into('<II',blob,18,0,0)
    assert study.prefix_epoch(blob)['epoch_candidate']==9


def test_metadata_gate_enforces_prefix_cap_and_never_infers_model_permission(tmp_path,monkeypatch):
    class Response(io.BytesIO):
        status=206
        headers={'Content-Type':'application/octet-stream','Content-Range':'bytes 0-99/1000'}
    monkeypatch.setattr(study.urllib.request,'urlopen',lambda *args,**kwargs:Response(b'x'*1000))
    cfg={'checkpoint_prefix_cap_bytes':100,'checkpoint_metadata_url':'https://example.invalid',
         'checkpoint_request_timeout_sec':1}
    result=study.checkpoint_gate(cfg,tmp_path)
    assert result['received_bytes']==100
    assert result['weights_license']=='unverified' and not result['locator_loading_allowed']
    assert not result['tensor_deserialization'] and result['epoch'] is None


def minimal_signature():
    cfg={'version':'v1','clip_length_sec':2.,'token_cap':32,'length_policy':'upstream_nonzero'}
    record={'video_sha256':'abc','duration_sec':12.,'width':320,'height':240}
    return study.signature(record,cfg,'weights',{'code':'sha'},{'torch':'version'})


@pytest.mark.parametrize('field',['pipeline_version','token_cap','length_policy','pixels','clip_weights_sha256',
                                  'source_sha256','versions','sampling','duration_sec','video_sha256'])
def test_official_cache_key_separates_every_contract_axis(field):
    original=minimal_signature();changed=copy.deepcopy(original);changed[field]='changed'
    assert study.digest_json(original)!=study.digest_json(changed)


def test_cache_rejects_byte_tamper_or_different_contract(tmp_path):
    sig=minimal_signature();tef,anchor=upstream_temporal_inputs(3,2.)
    np.savez(tmp_path/'features.npz',images=np.eye(3,dtype=np.float32),times=np.arange(3)*2.,
             tef=tef.numpy(),anchors=anchor.numpy())
    meta={'signature':sig,'cache_key':study.digest_json(sig),
          'files_sha256':{'features.npz':sha256_file(tmp_path/'features.npz')}}
    write_json_atomic(tmp_path/'metadata.json',meta)
    study.validate_cache(tmp_path,sig)
    with pytest.raises(ValueError,match='contract'):
        study.validate_cache(tmp_path,{**sig,'token_cap':77})
    with (tmp_path/'features.npz').open('ab') as file:file.write(b'tamper')
    with pytest.raises(ValueError,match='bytes'):study.validate_cache(tmp_path,sig)


def test_generated_video_keeps_logical_seconds_and_ambiguous_source_candidates(demo,tmp_path,monkeypatch):
    monkeypatch.setattr(study,'checked_output',lambda path:Path(path))
    record={'video':demo['video'],'width':320,'height':240,'duration_sec':60.}
    pixels,grid,phase=study.decode_pixels(record,tmp_path,40)
    assert pixels.shape==(30,224,224,3) and np.array_equal(grid,np.arange(30)*2.)
    assert all(p['source_pts_candidates_sec'] for p in phase)
    assert any(not p['unique_source_pts'] for p in phase)
    with pytest.raises(ValueError,match='budget'):study.decode_pixels(record,tmp_path,2)


def test_inference_ignores_labels_and_failed_video_retains_empty_query(tmp_path):
    old=tmp_path/'old.npz';new=tmp_path/'new';new.mkdir()
    features=np.array([[1.,0],[0.,1],[1.,0]],dtype=np.float32)
    np.savez(old,images=features,times=[0.,4.,8.])
    np.savez(new/'features.npz',images=features,times=[0.,2.,4.])
    records=[{'qid':1,'video_sha256':'present','images_path':str(old),'duration_sec':12.},
             {'qid':2,'video_sha256':'missing','images_path':str(old),'duration_sec':12.}]
    text={'pool_1':np.array([1.,0]),'pool_2':np.array([0.,1])}
    index={'present':{'folder':str(new)}}
    original=study.predict_inputs(records,text,index)
    altered=[{**r,'relevant_windows':[[0.,12.]],'duration_label':-1} for r in records]
    assert study.predict_inputs(altered,text,index)==original
    assert original[1][2]==[] and original[2][2]==[] and len(original[1])==2
    for predictions in original[:2]:
        for windows in predictions.values():
            assert all(isinstance(a,float) and 0<=a<b<=12. for a,b,_ in windows)


def test_outputs_refuse_path_traversal():
    with pytest.raises(ValueError,match='outputs'):study.checked_output(study.ROOT/'outputs/../escape')
