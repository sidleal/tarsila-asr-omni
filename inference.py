import torch
from datasets import load_dataset
# Note: The import name might differ from the package name, 
# but for this specific lib it is usually 'omnilingual_asr'
import omnilingual_asr 
import re
import jiwer
import jiwer.transforms as tr
import csv
from tqdm.auto import tqdm
    
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch
import torchaudio


from datasets import load_dataset

def main():
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    # Simple dataset test
    try:
        ds = load_dataset("glue", "mrpc", split="train[:10]")
        print(f"✅ Datasets loaded: {len(ds)} rows")
    except Exception as e:
        print(f"❌ Datasets error: {e}")

    print("✅ Omnilingual ASR imported successfully")

    
    cer_transform = tr.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )
    
    # It's the jiwer default transform
    wer_transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    
    def compute_cer(reference, hypothesis):
        reference = reference.lower()
        hypothesis = hypothesis.lower()
        cer = jiwer.wer(reference, hypothesis, reference_transform =cer_transform, hypothesis_transform=cer_transform)
        return cer
    
    def compute_wer(reference, hypothesis):
        reference = reference.lower()
        hypothesis = hypothesis.lower()
        wer = jiwer.wer(reference, hypothesis, reference_transform =wer_transform, hypothesis_transform=wer_transform)
        return wer
    
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõũúû1234567890%\-\n/\\ "
    
    def replace_special_tokens_and_normalize(text):
        text = text.lower()
    
        map_words = {
            "éh": "eh",
            "ehm": "eh",
            "ehn": "eh",
            "hum": "uh",
            "hm": "uh",
            "uhm": "uh",
            "hã": "ah",
            "ãh": "ah",
            "ã":  "ah",
            "hmm": "uh",
            "mm": "uh",
            "mhm": "uh"
        }
    
        text = re.sub("h+", "h", text)
        text = re.sub("[^{}]".format(alphabet+" "), " ", text)
        text = re.sub("[ ]+", " ", text)
    
        words = text.split(' ')
        new_words = []
        for word in words:
            if word == '' or word == ' ':
                continue
            if word in map_words:
                new_words.append(map_words[word])
            else:
                new_words.append(word)
    
        return " ".join(new_words)
    
    def calculate_wer_cer(reference, hypothesis):
        if reference.strip() == '' or hypothesis.strip() == '':
            return 1, 1
        wer = compute_wer(reference, hypothesis)
        cer = compute_cer(reference, hypothesis)
        return wer, cer
    
    

    from fairseq2.assets import AssetCard

    model_300M_4k = AssetCard("omniASR_LLM_300M_Tarsila_4k", {
        "model_family": "wav2vec2_llama",  
        "model_arch": "300m",   
        "checkpoint": "file:///home/jovyan/omnilingual-asr/output/ws_1.96866555/checkpoints/step_4000/model/pp_00/tp_00/sdp_00.pt",
        "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
        "tokenizer_family": "char_tokenizer",
    })

    model_300M_9k = AssetCard("omniASR_LLM_300M_Tarsila_9k", {
        "model_family": "wav2vec2_llama",  
        "model_arch": "300m",   
        "checkpoint": "file:///home/jovyan/omnilingual-asr/output/ws_1.96866555/checkpoints/step_9000/model/pp_00/tp_00/sdp_00.pt",
        "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
        "tokenizer_family": "char_tokenizer",
    })

    model_1B_4k = AssetCard("omniASR_LLM_1B_Tarsila_4k", {
        "model_family": "wav2vec2_llama",  
        "model_arch": "1b",   
        "checkpoint": "file:///home/jovyan/omnilingual-asr/output/ws_1.0b418119/checkpoints/step_4000/model/pp_00/tp_00/sdp_00.pt",
        "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
        "tokenizer_family": "char_tokenizer",
    })

    model_1B_9k = AssetCard("omniASR_LLM_1B_Tarsila_9k", {
        "model_family": "wav2vec2_llama",  
        "model_arch": "1b",   
        "checkpoint": "file:///home/jovyan/omnilingual-asr/output/ws_1.0b418119/checkpoints/step_9000/model/pp_00/tp_00/sdp_00.pt",
        "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
        "tokenizer_family": "char_tokenizer",
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print("load model")
    pipeline1 = ASRInferencePipeline(model_card=model_300M_4k, device=device, dtype=torch_dtype)
    pipeline2 = ASRInferencePipeline(model_card=model_300M_9k, device=device, dtype=torch_dtype)
    pipeline3 = ASRInferencePipeline(model_card=model_1B_4k, device=device, dtype=torch_dtype)
    pipeline4 = ASRInferencePipeline(model_card=model_1B_9k, device=device, dtype=torch_dtype)

    print("models ok")
    

    def trancript_with_omni(audio_tensor):
        import io
        import soundfile as sf
        import numpy as np
        
        sample_rate = 16000
        
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format='WAV')
        buf.seek(0)
        raw_uint8_data = np.frombuffer(buf.read(), dtype=np.uint8)
        
        audio_files = [raw_uint8_data]
        lang = ["por_Latn"]

        return [trancript_pipe_1(audio_files, lang), trancript_pipe_2(audio_files, lang), trancript_pipe_3(audio_files, lang), trancript_pipe_4(audio_files, lang)]


    def trancript_pipe_1(audio_files, lang):
        transcriptions = pipeline1.transcribe(audio_files, lang=lang, batch_size=1)
        return transcriptions[0]

    def trancript_pipe_2(audio_files, lang):
        transcriptions = pipeline2.transcribe(audio_files, lang=lang, batch_size=1)
        return transcriptions[0]

    def trancript_pipe_3(audio_files, lang):
        transcriptions = pipeline3.transcribe(audio_files, lang=lang, batch_size=1)
        return transcriptions[0]

    def trancript_pipe_4(audio_files, lang):
        transcriptions = pipeline4.transcribe(audio_files, lang=lang, batch_size=1)
        return transcriptions[0]


    def inference_and_calc_wer_cer(text_orig_norm, audio_array):
        output_asr = trancript_with_omni(audio_array)
        ret = []
        for it in output_asr:
            output_asr_norm = replace_special_tokens_and_normalize(it)
            wer, cer = calculate_wer_cer(text_orig_norm, output_asr_norm)
            ret.append([it, output_asr_norm, wer, cer])
        return ret
    
    dataset_link = "sidleal/TARSILA-ASR-V1"
    
    tarsila_test = load_dataset(dataset_link, split="test", streaming=True)
    
    print(tarsila_test)
    
    file_name = 'tarsila_asr_test_inference_omni_tuned.csv'
    
    i=0
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        csv_data = [
            "idx","origin","gender","duration","ref","ref_norm",
            "omni_300M_4k","omni_300M_4k_norm","omni_300M_4k_wer","omni_300M_4k_cer",
            "omni_300M_9k","omni_300M_9k_norm","omni_300M_9k_wer","omni_300M_9k_cer",
            "omni_1B_4k","omni_1B_4k_norm","omni_1B_4k_wer","omni_1B_4k_cer",
            "omni_1B_9k","omni_1B_9k_norm","omni_1B_9k_wer","omni_1B_9k_cer"
        ]
        writer.writerow(csv_data)
    
        for item in tqdm(tarsila_test, desc=f"Processing test set"):
            print(item)
            #if i < 10833:
            #    i+=1
            #    continue
            if i > 5:
                break
                
            audio_array=item["audio"]["array"]
            text_orig = item["text"]
            text_orig_norm = replace_special_tokens_and_normalize(text_orig)
    
            #try:
            ret_asr = inference_and_calc_wer_cer(text_orig_norm, audio_array)
            output_asr_1, output_asr_norm_1, wer_1, cer_1 = ret_asr[0]
            output_asr_2, output_asr_norm_2, wer_2, cer_2 = ret_asr[1]
            output_asr_3, output_asr_norm_3, wer_3, cer_3 = ret_asr[2]
            output_asr_4, output_asr_norm_4, wer_4, cer_4 = ret_asr[3]
                
            csv_data = [
                i, item["origin"], item["gender"][0], item["duration"], text_orig, text_orig_norm,
                output_asr_1, output_asr_norm_1, wer_1, cer_1,
                output_asr_2, output_asr_norm_2, wer_2, cer_2,
                output_asr_3, output_asr_norm_3, wer_3, cer_3,
                output_asr_4, output_asr_norm_4, wer_4, cer_4,
            ]
            writer.writerow(csv_data)
            #except Exception as e:
            #    print(e)
            
            i+=1




if __name__ == "__main__":
    main()
