import sentencepiece as spm

# Setup the command with all parameters including special token IDs
spm_command = '--input=cleaned_combined_data.txt --model_prefix=m_translation ' \
              '--vocab_size=135000 --model_type=bpe ' \
              '--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --character_coverage=0.9900 '

# Train the model
spm.SentencePieceTrainer.train(spm_command)