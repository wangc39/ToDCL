import numpy as np
from collections import Counter

class F1Metrics():
  def __init__(self):
    pass

  # Build word and bigram frequencies.
  def _build_distro(self, distro, candidates, vocab=None, probs=False):
    for text in candidates:
      words = text.split()
      word_count = len(words)
      for i, word in enumerate(words):
        if vocab:
          word = word if vocab.get(word) else '<unk>' #TODO decide whether to change it to <unk>
        w_in_dict = distro['uni'].get(word)
        distro['uni'][word] = distro['uni'][word] + 1 if w_in_dict else 1

        # Bigrams.
        if i < word_count - 1:
          word2 = words[i + 1]
          if vocab:
            word2 = words[i + 1] if vocab.get(words[i + 1]) else '<unk>'
          bi = (word, word2)
          bigram_in_dict = distro['bi'].get(bi)
          distro['bi'][bi] = distro['bi'][bi] + 1 if bigram_in_dict else 1

    if probs:
      distro['uni'] = self._convert_to_probs(distro['uni'])
      distro['bi'] = self._convert_to_probs(distro['bi'])

  @staticmethod
  def compute_f1(match, predictions, total):
      P = match / predictions
      R = match / total
      f1 = float('nan') if P==0 and R==0 else 2*P*R/(P+R)
      return f1

  def calculate_metrics(self, candidates, references):
      '''
      candidates: [[['That', 'is', 'right'], ['regard', 'as']], [['I', 'pray', 'it'], ['go', 'hell']]]
      references: [['I', 'love', 'you'], ['let', 'us', 'go']]
      return: [['我', '很想', '你', '啊'], ['静静'], ['这', '座', '城市', '没有', '你']]
      '''

      # print(candidates)
      # print("====="*20)
      # print(references)
      
      candidates = [[' '.join(reply) for reply in item] for item in candidates]
      references = [' '.join(item[0]) for item in references]

      # candidates: [['You are welcome!', 'Thank you!', 'That is all right']]
      # references: ['I like you very much', 'I love you']

      # print(candidates)
      # print("====="*20)
      # print(references)


      assert len(candidates) == len(references)

      f1s = []
      for i in range(len(candidates[0])):
        golden_char_total = 0.0
        pred_char_total = 0.0
        hit_char_total = 0.0
        for _, (candss, golden_response) in enumerate(zip(candidates, references)):
          response = candss[i]
          # print(candss)
          # exit(1)
          
          # print(response)
          # print(response.split())
          # print("".join(response.split()))
          # break

          # common = Counter(response.split()) & Counter(golden_response.split())
          common = Counter(response.split()) & Counter(golden_response.split())

          hit_char_total += sum(common.values())
          golden_char_total += len(golden_response.split())
          pred_char_total += len(response.split())

        f1s.append(self.compute_f1(hit_char_total, pred_char_total, golden_char_total))

      avg_f1 = np.mean(f1s)
      max_f1 = np.max(f1s)
      return avg_f1, max_f1