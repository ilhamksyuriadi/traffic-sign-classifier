# In your notebook or create check_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model('model/traffic_sign_classifier.h5')

# Create validation generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'data/Train',
    target_size=(48, 48),
    batch_size=128,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(val_generator)
print(f"\nOverall Validation Accuracy: {accuracy:.2%}")