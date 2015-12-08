
import shlex
import subprocess
import threading

from sklearn.base import BaseEstimator

from senti.utils import reiterable
from senti.utils.sklearn_ import EmptyFitMixin

__all__ = ['ArkTweetPosTagger']


class ArkTweetPosTagger(BaseEstimator, EmptyFitMixin):
    # from runTagger.sh
    RUN_TAGGER_CMD = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar ../../third/ark-tweet-nlp/ark-tweet-nlp-with-deps.jar'

    @staticmethod
    def _write_input(docs, proc, buffer_sema):
        for doc in docs:
            buffer_sema.acquire()
            # remove carriage returns as they are tweet separators for the stdin interface
            proc.stdin.write((doc.replace('\n', ' ') + '\n').encode('utf-8'))
            proc.stdin.flush()
        proc.stdin.close()

    @reiterable
    def transform(self, docs, buffer_size=100):
        args = shlex.split(self.RUN_TAGGER_CMD) + ['--output-format', 'conll']
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buffer_sema = threading.BoundedSemaphore(buffer_size)
        t = threading.Thread(target=self._write_input, args=(docs, proc, buffer_sema))
        t.start()
        while True:
            # reading can only follow writing unless EOF is reached so buffer_sema >= 0
            res = []
            while True:
                line = proc.stdout.readline().decode('utf-8').rstrip()
                if line == '':
                    break
                word, tag, confidence = line.split('\t')
                res.append((word, tag, float(confidence)))
            if not res:
                break
            yield res
            buffer_sema.release()
        t.join()
