
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, keep_verbose_stats=False):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = [] if keep_verbose_stats else None

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        if self.data is not None:
            self.data.append([val, n])
