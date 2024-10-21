import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
import pickle

# Getting Data
file_path = 'medical_dataset.csv'
df = pd.read_csv(file_path,encoding='latin-1')
print(df.head())

def clean_text_func(text):
    
    text = str(text) 
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    return text

df['Description'] = df['Description'].apply(lambda x: clean_text_func(x))

tfidf = TfidfVectorizer(stop_words="english")  
X = tfidf.fit_transform(df['Description']).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Disease'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# With ANN:-

model = Sequential()

# Input layer and two hidden layers with Dropout to prevent overfitting
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))  # Dropout to prevent overfitting

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.1))

# Output layer (classification problem, we'll use softmax)
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1)


model.save('model.keras')
#pickle.dump(model,open("model.pkl","wb"))
pickle.dump(tfidf,open("vectorizer.pkl","wb"))
pickle.dump(label_encoder,open("label_encoder.pkl", "wb"))

import os
os.getcwd()