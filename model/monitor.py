
class monitor_lr(object):
    # adaptively change the learning rate based on the validation result
    def __init__(self, step_bin=3, step_wait=5, thres=0.95, step_max=100):
        # step_bin: how many validation needed for one point
        # step_wait: how many points consecutively to fail to decrease enough before changing the learning rate
        # thres: threshold of learning rate to be changed
        # step_max: if reached the max number, decrease the learning rate
        # 1. bin the validation results for robust statistics
        # 2. stopping criteria: 
        self.step_bin = step_bin
        self.step_wait = step_wait
        self.thres = thres
        self.step_max = step_max
        self.num_change = 0

        self.reset()

    def add(self,result):
        self.val_result.append(result)
        self.val_id += 1
        if self.val_id % self.step_bin == 0:
            self.val_stat.append(sum(self.val_result[-self.step_bin:])/float(self.step_bin))

    def toChange(self):
        change = False
        if self.val_id>self.step_max:
            change = True
        elif len(self.val_stat)>self.step_wait and self.val_id % self.step_bin == 0 \
                and min(self.val_result[-self.step_wait:])>min(self.val_result[:-self.step_wait])*self.thres:
            change = True
        
        if change:
            self.num_change += 1
            self.reset()
        return change

    def reset(self):
        self.val_id = 0
        self.val_result = []
        self.val_stat = []
        self.change = False

