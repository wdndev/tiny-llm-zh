import sentencepiece as spm
import os
import glob

def tain_chinses_spm(input_txt_dir, vocab_size, output_dir="."):
    # 保存的模型名称
    prefix = os.path.join(output_dir, f"test_chinese_spm_{vocab_size}")

    text_filenames = sorted(glob.glob(os.path.join(input_txt_dir, "*.txt")))
    print("file list: ", text_filenames)

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=text_filenames,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=0.9995,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,       # 是否将数字划分为单个 token, 在 llama 中是这么做的
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   max_sentence_length=24000)


    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

def test_chinese_spm(spm_model_path):
    sp_bpe = spm.SentencePieceProcessor() 
    sp_bpe.load(spm_model_path)
    print('*** BPE ***')
    print(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎'))
    print(len(sp_bpe.encode_as_pieces('翻译下面的句子为英文：有朋自远方来，不亦乐乎')))


if __name__ == "__main__":
    input_txt_dir = "baike_txt"
    vocab_size = 20000
    output_dir = "sp_output"
    tain_chinses_spm(input_txt_dir, vocab_size, output_dir)

    # test_chinese_spm("sp_output/chinese_spm_20000.model")