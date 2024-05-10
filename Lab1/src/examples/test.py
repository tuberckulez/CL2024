import torch
import nemo.collections.asr as nemo_asr

 

def infer_greedy(files, asr_model):
        transcripts = asr_model.transcribe(files, batch_size=20)
        return transcripts


if __name__ == '__main__':
        model = "../../models/QuartzNet15x5_golos.nemo"
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model)
        files = ["../../data/001ce26c07c20eaa0d666b824c6c6924.wav"]
        hyps = infer_greedy(files, asr_model)
        print(hyps)