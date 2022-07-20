import torch



def get_accuray(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    """ 
    get accuracy
    Args:
        SR ([type]): Segmentation Result
        GT ([type]): Ground Truth
        threshold (float, optional): threshold. Defaults to 0.5.
    """
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    SR = SR > threshold  # binarization -> Tensor[bool]
    # torch.max(input:Tensor)会返回最大的一个值，所以最后GT中只有一个值为True，其余全为False
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    """ Sensitivity == Recall """
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP: True Positive
    # FN: False Negative
    TP = ((SR == 1).long() + (GT == 1).long()) == 2
    FN = ((SR == 0).long() + (GT == 1).long()) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).long() + (GT == 0).long()) == 2
    FP = ((SR == 1).long() + (GT == 0).long()) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    
    return SP


def get_precision(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).long() + (GT == 1).long()) == 2
    FP = ((SR == 1).long() + (GT == 0).long()) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1

def get_JS(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR.long() + GT.long()) == 2)
    Union = torch.sum((SR.long() + GT.long()) >= 1)
    
    JS = float(Inter) / (float(Union) + 1e-6)
    
    return JS

def get_DC(SR: torch.Tensor, GT: torch.Tensor, threshold=0.5):
    
    if SR.size(1) != 1:
        SR = SR[:, 1:, :, :]
        GT = GT[:, 1:, :, :]
    
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.long() + GT.long()) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC
