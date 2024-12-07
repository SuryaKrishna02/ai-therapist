import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer

def inference():
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    model, processor, tokenizer = model_init(model_path)

    # Audio-visual Inference
    audio_video_path = "data/clip_001.mp4"
    preprocess = processor["video"]
    audio_video_tensor = preprocess(audio_video_path, va=True)
    question = "How many woman are present in the video and what is the emotion of each of the two women"

    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=False,
    )

    print(output)


if __name__ == "__main__":
    inference()