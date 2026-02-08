import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("SAME TRAINING AS BEFORE + AUTO-SAVE PROTECTION")
print("="*60)

# Paths
train_dir = 'Train'
val_dir = 'Validation'
test_dir = 'Test'

IMG_SIZE = 224
BATCH_SIZE = 32  # Same as before

# Data generators - EXACT SAME AS YOUR SUCCESSFUL RUN
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_gen = val_test_datagen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)
test_gen = val_test_datagen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Class weights
cw = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
cw_dict = dict(enumerate(cw))

print("Classes:", train_gen.class_indices)

# Model - EXACT SAME ARCHITECTURE
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(8, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nParameters: {model.count_params():,}")

# *** CRASH PROTECTION - SAVES AFTER EVERY EPOCH ***
checkpoint = ModelCheckpoint(
    'iesa_checkpoint_epoch_{epoch:02d}_val{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=False,  # Save EVERY epoch
    verbose=1
)

best_checkpoint = ModelCheckpoint(
    'iesa_model_BEST.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# PHASE 1: Frozen base - 10 EPOCHS (same as your successful run)
print("\nPHASE 1: Training frozen base (10 epochs)")
print("="*60)

h1 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=cw_dict,
    callbacks=[checkpoint, best_checkpoint],
    verbose=1
)

print("\n✓ PHASE 1 COMPLETE! Model auto-saved!")

# PHASE 2: Fine-tuning - 10 EPOCHS
print("\nPHASE 2: Fine-tuning all layers (10 epochs)")
print("="*60)

base.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

h2 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=cw_dict,
    callbacks=[checkpoint, best_checkpoint],
    verbose=1
)

print("\n✓ PHASE 2 COMPLETE!")

# Load best model
model = tf.keras.models.load_model('iesa_model_BEST.h5')
model.save('iesa_model_final.h5')

# Test
loss, acc = model.evaluate(test_gen)
print(f"\n*** FINAL TEST ACCURACY: {acc*100:.2f}% ***")

# Classification report
test_gen.reset()
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

# Convert to ONNX
print("\nConverting to ONNX...")
import tf2onnx, onnx
input_sig = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_sig, opset=13)
onnx.save(onnx_model, "iesa_model.onnx")

import os
size = os.path.getsize("iesa_model.onnx") / (1024 * 1024)
print(f"✓ ONNX: {size:.2f} MB")

# Download
from google.colab import files
files.download('iesa_model.onnx')
files.download('iesa_model_BEST.h5')
files.download('iesa_model_final.h5')

print("\n✅ COMPLETE! Files downloading!")
