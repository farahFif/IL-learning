import torch
class  aLoader(torch.utils.data.Dataset):
    def __init__(self, dataset,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = dataset
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index][1]
        labels = self.data[index][0]

        sample = (image, labels) #or simply sample = self.data[i]
        if self.transform:
            sample = (self.transform(image), labels)

        return sample