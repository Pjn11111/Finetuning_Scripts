import json
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import time

# Custom callback for live loss tracking
class LossTracker(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            elapsed_time = time.time() - self.start_time
            print(f"\033[92mStep {state.global_step}: Loss = {logs['loss']:.4f}, Time = {elapsed_time:.2f}s\033[0m")
            self.losses.append((state.global_step, logs['loss'], elapsed_time))

    def plot(self):
        steps, losses, times = zip(*self.losses)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss over Training Steps')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(steps, times, label='Time')
        plt.xlabel('Step')
        plt.ylabel('Time (s)')
        plt.title('Time per Training Step')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()

# ASCII Art for a warm welcome
print(r"""
__ | | |(_)| | | __
""")
print("Welcome to the BERT fine-tuning script!\n")

# Get user input for epochs and learning rate
epochs = int(input("Enter the number of training epochs: "))
learning_rate = float(input("Enter the learning rate: "))

# Load the JSON dataset
print("Loading dataset...")
with open('/home/user/Desktop/Poojan/100ent.json', 'r') as f:
    data = json.load(f)

# Convert the data into a list of dictionaries suitable for Hugging Face datasets
print("Formatting dataset...")
formatted_data = [
    {
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]["Description"]  # Use a relevant field from the "output" for labels
    }
    for item in data
]

# Extract unique labels
unique_labels = set(item["output"] for item in formatted_data)
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

# Update dataset with labels
for item in formatted_data:
    item["label"] = label_to_id[item["output"]]

# Create a Hugging Face Dataset
print("Creating Dataset object...")
dataset = Dataset.from_list(formatted_data)

# Initialize the tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    combined_inputs = [inst + " " + inp for inst, inp in zip(examples['instruction'], examples['input'])]
    model_inputs = tokenizer(combined_inputs, padding='max_length', truncation=True, max_length=128)
    
    labels = [label_to_id.get(desc, 0) for desc in examples['output']]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load the pretrained BERT model
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_labels))

# Set up the training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=epochs,         # total number of training epochs (set by user)
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',     # evaluate every logging_steps
    save_strategy='steps',           # save model every logging_steps
    learning_rate=learning_rate      # learning rate (set by user)
)

# Initialize the Trainer with a custom callback
print("Initializing Trainer...")
loss_tracker = LossTracker()
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_dataset,     # training dataset
    eval_dataset=tokenized_dataset,      # evaluation dataset
    callbacks=[loss_tracker]             # Add custom callback for live loss tracking
)

# Fine-tune the model
print("Starting training...")
trainer.train()

# Plot and save training metrics
print("Plotting training metrics...")
loss_tracker.plot()

# Evaluate the model
print("Evaluating the model...")
trainer.evaluate()

# Save the fine-tuned model
print("Saving the model...")
model.save_pretrained('./my_finetuned_model')
print("Model saved successfully!")

