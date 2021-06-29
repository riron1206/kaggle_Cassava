import os
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class WrapTensorboard(object):
    """
    (wrapper) Tensorboard
    """
    def __init__(self, log_dir='./'):
        super(WrapTensorboard, self).__init__()
        self.writer = None
        if log_dir is not None and os.path.exists(log_dir):
            self.writer = SummaryWriter(log_dir=log_dir)

    def __del__(self):
        self.close()

    def open(self, log_dir):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            raise UserWarning('TensorBoard(SummryWriter)は既に設定されています.')

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def writeScalar(self, keyword, x, value):
        if self.writer is not None:
            if type(value) == dict:
                self.writer.add_scalars(keyword, value, x)
            else:
                self.writer.add_scalar(keyword, value, x)

class WrapTqdm(object):
    """
    (wrapper) Tqdm
    """
    def __init__(self, run='local', **kwargs):
        super(WrapTqdm, self).__init__()
        if run is None or run.lower() == 'local':
            self.instance = tqdm(**kwargs)
        else:
            self.instance = None

    def __del__(self):
        self.close()

    def set_description(self, text):
        if self.instance is not None:
            self.instance.set_description(text, refresh=True)

    def update(self):
        if self.instance is not None:
            self.instance.update()

    def close(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None

class WrapStdout(object):
    """
    (wrapper) stdout
    """
    def __init__(self, filepath='stdout.txt'):
        super(WrapStdout, self).__init__()
        self.filepath = filepath
        self.textbuff = [ ]
        self.indtbuff = ''

    def __call__(self, text):
        self.print(text)

    def __del__(self):
        self.flush()

    def flush(self):
        if len(self.textbuff) > 0:
            if self.filepath is not None:
                with open(self.filepath, mode='a', encoding='utf-8') as f:
                    for t in self.textbuff:
                        f.write('{}\n'.format(t))
            self.textbuff = [ ]

    def close(self):
        self.flush()

    def setIndent(self):
        self.indtbuff += '    '

    def endIndent(self):
        if len(self.indtbuff) >= 4:
            self.indtbuff = self.indtbuff[:-4]

    def print(self, text):
        logstr = '[{}] {}{}'.format(self._inner_timestamp_(), self.indtbuff, text)
        self.textbuff.append(logstr)
        print(logstr)

    def printLine(self):
        logstr = '[{}] ------------------------------------------------------------'.format(self._inner_timestamp_())
        self.textbuff.append(logstr)
        print(logstr)

    def writeInfo(self, text):
        logstr = '[{}] {}{}'.format(self._inner_timestamp_(), self.indtbuff, text)
        self.textbuff.append(logstr)

    def _inner_timestamp_(self):
        t = datetime.datetime.now()
        return t.strftime('%Y/%m/%d %H:%M:%S')
