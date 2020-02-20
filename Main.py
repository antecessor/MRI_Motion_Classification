from DeepLearning.CNNModel import lstmTrain
from Utils.DataUtils.DataLoader import Dataloder
from Utils.DataUtils.Dataset import Dataset

trainDataset = Dataset(train=True)
testDataset = Dataset(train=False)

batch_size = 30
train_dataloader = Dataloder(trainDataset, batch_size=30, shuffle=True)
test_dataloader = Dataloder(testDataset, batch_size=30, shuffle=True)

lstmTrain(train_dataloader, test_dataloader, 5)
