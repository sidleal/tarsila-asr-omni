from huggingface_hub import HfApi, ModelCard, ModelCardData

repo_id = "sidleal/omniASR_LLM_300M_Tarsila_4k"
local_folder = "./output/ws_1.96866555/checkpoints/step_4000/model/pp_00/tp_00"

api = HfApi()

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

card_data = ModelCardData(
    language="pt",
    license="mit",
    library_name="fairseq2",
    tags=["automatic-speech-recognition", "audio", "omnilingual-asr"]
)
card = ModelCard.from_template(
    card_data,
    model_id=repo_id,
    model_description="ASR model finetuned from Omnilingual ASR 300M."
)
card.save(f"{local_folder}/README.md")

print(f"Uploading files to {repo_id}...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    commit_message="Initial upload of Omnilingual ASR Tarsila model"
)
print("Upload complete!")
