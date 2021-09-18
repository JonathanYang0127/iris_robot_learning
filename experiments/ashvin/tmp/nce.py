import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

temperature = 1
cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

out0 = torch.randn(10, 4)
out1 = 1 * out0 + 0.4 * torch.randn(10, 4) # out 0

z0 = out0
batch_size = z0.shape[0]
z1 = out1
z0 = torch.nn.functional.normalize(z0, dim=1)
z1 = torch.nn.functional.normalize(z1, dim=1)
output = torch.cat((z0, z1), dim=0)
logits = torch.einsum("nc,mc->nm", output, output) / temperature
print("einsum")
print(logits)
logits = logits[~torch.eye(2*batch_size,device=z0.device).byte()].view(2*batch_size, -1)
print("reshape")
print(logits)
labels = torch.arange(batch_size, device=z0.device, dtype=torch.long)
labels = torch.cat([labels + batch_size - 1, labels])
print(labels)
loss = cross_entropy(logits, labels)
print(loss)

print(accuracy(logits, labels, (1, )))
