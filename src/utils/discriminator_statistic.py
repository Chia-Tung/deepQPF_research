import torch

def compute_auroc(generated_data_probab, actual_data_probab):
    """This function is not been used now"""
#     auroc = AUROC(num_classes=2)
    pred = torch.cat([generated_data_probab, actual_data_probab])
    tar = torch.zeros_like(pred)
    tar[len(generated_data_probab):] = 1
#     return auroc(pred, tar.to(torch.int))
    return 1

class DiscriminatorStats:
    def __init__(self):
        self._genP = None
        self._actP = None
        self.reset()

    def reset(self):
        self._genP = []
        self._actP = []

    def update(self, generated_data_probablity, actual_data_probablity):
        self._genP.append(generated_data_probablity.cpu().view(-1, ))
        self._actP.append(actual_data_probablity.cpu().view(-1, ))

    def get(self, threshold=0.5):
        neg = torch.cat(self._genP)
        pos = torch.cat(self._actP)
        assert len(neg) == len(pos)
        return {
            'auc': compute_auroc(neg, pos),
            'pos_accuracy': torch.mean((pos >= threshold).double()),
            'neg_accuracy': torch.mean((neg < threshold).double()),
            'N': len(pos),
        }
    
    def raw_data(self):
            return {'actual': torch.cat(self._actP), 'generated': torch.cat(self._genP)}

    def __len__(self):
        return sum([len(x) for x in self._genP])