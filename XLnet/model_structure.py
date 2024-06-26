from transformers import XLNetModel, XLNetConfig

def print_model_structure(model):
    # Print model architecture
    print(model)

# Load XLNet model configuration
config = XLNetConfig.from_pretrained('xlnet-base-cased')

# Instantiate XLNet model
model = XLNetModel(config)

# Print the model structure
print_model_structure(model)


