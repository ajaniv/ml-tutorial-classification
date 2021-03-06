from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
input_classes = ['suzuki', 'ford', 'suzuki', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
print ("class mapping:")
for i, item in enumerate(label_encoder.classes_):
    print (item, '-->', i)
    
# encode
labels = ['toyota', 'ford', 'suzuki']
encoded_labels = label_encoder.transform(labels)
print ("Labels:", labels)
print ("Encoded labels:", list(encoded_labels))

# decode
encoded_labels = [3, 2, 0, 2, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print ("Encoded labels:", encoded_labels)
print ("Decoded labels:", list(decoded_labels))

"""
class mapping:
bmw --> 0
ford --> 1
suzuki --> 2
toyota --> 3
Labels: ['toyota', 'ford', 'suzuki']
Encoded labels: [3, 1, 2]
Encoded labels: [3, 2, 0, 2, 1]
Decoded labels: ['toyota', 'suzuki', 'bmw', 'suzuki', 'ford']
"""