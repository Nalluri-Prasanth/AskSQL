import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import os

# Configuration
MODEL_NAME = 'cssupport/t5-small-awesome-text-to-sql'
OUTPUT_DIR = 'models/avgs_t5_wikisql'
DATA_DIR = 'data'



BATCH_SIZE = 8
EPOCHS = 3 # Increased for robust learning of new patterns
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

class WikiSQLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_len, max_target_len):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text = item['input_text']
        target_text = item['target_text']

        # Tokenize
        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_len, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text, 
                max_length=self.max_target_len, 
                padding='max_length', 
                truncation=True,
                return_tensors="pt"
            )

        return {
            'input_ids': model_inputs['input_ids'].flatten(),
            'attention_mask': model_inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()
        }

def train():
    print("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Apply LoRA
    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_path = os.path.join(DATA_DIR, 'train_processed.csv')
    dev_path = os.path.join(DATA_DIR, 'dev_processed.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed training data not found at {train_path}. Run scripts/generate_synthetic.py first.")
        
    train_dataset = WikiSQLDataset(train_path, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = WikiSQLDataset(dev_path, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH) if os.path.exists(dev_path) else None

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,

        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        save_total_limit=2,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}...")
    # Merge LoRA weights back into the base model for easier inference
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train()
