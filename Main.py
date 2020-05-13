from DeepLearning.CNNModel import CNNTrain
from Utils.DataUtils.DataLoader import Dataloder
from Utils.DataUtils.Dataset import Dataset

trainDataset = Dataset(train="train")
testDataset = Dataset(train="test")
validationDataset = Dataset(train="validation")

batch_size = 16
train_dataloader = Dataloder(trainDataset, batch_size=batch_size, shuffle=True)
test_dataloader = Dataloder(testDataset, batch_size=batch_size, shuffle=True)
validation_dataloader = Dataloder(validationDataset, batch_size=batch_size, shuffle=True)

trainedModel = CNNTrain(train_dataloader, test_dataloader, validation_dataloader, 5, show=True)
trainedModel.save("motionClassificationModel.h5")
