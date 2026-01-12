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
    
    
    

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print("load model")
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_300M_Tarsila_4k", device=device, dtype=torch_dtype)
    #pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B", device=device, dtype=torch_dtype)
    
    print("model ok")
    
    def trancript_with_omni(audio_tensor):
        audio_files = [audio_tensor]
        lang = ["por_Latn"]
        transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=1)
        return transcriptions[0]
    
    def inference_and_calc_wer_cer(text_orig_norm, audio_array):
        output_asr = trancript_with_omni(audio_array)
        output_asr_norm = replace_special_tokens_and_normalize(output_asr)
        wer, cer = calculate_wer_cer(text_orig_norm, output_asr_norm)
        return output_asr, output_asr_norm, wer, cer
    
    dataset_link = "sidleal/TARSILA-ASR-V1"
    
    tarsila_test = load_dataset(dataset_link, split="test", streaming=True)
    
    print(tarsila_test)
    
    file_name = 'tarsila_asr_test_inference_omni_tuned.csv'
    
    i=0
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        csv_data = [
            "idx","origin","gender","duration","ref","ref_norm",
            "omni_300M_4k","omni_300M_4k_norm","omni_300M_4k_wer","omni_300M_4k_cer"
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
    
            try:
                output_asr_1, output_asr_norm_1, wer_1, cer_1 = inference_and_calc_wer_cer(text_orig_norm, audio_array)
        
                csv_data = [
                    i, item["origin"], item["gender"][0], item["duration"], text_orig, text_orig_norm,
                    output_asr_1, output_asr_norm_1, wer_1, cer_1
                ]
                writer.writerow(csv_data)
            except Exception as e:
                print(e)
            
            i+=1




if __name__ == "__main__":
    main()
