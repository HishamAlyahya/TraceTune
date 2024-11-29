import os
import random

from chaintune import get_traced_sample
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune

from dotenv import load_dotenv

load_dotenv()

# ports = [7140, 7141, 7142, 7143, 7144, 7145]
# llamaChat = dspy.HFClientTGI(model="meta-llama/Llama-2-13b-chat-hf", port=ports, max_tokens=150)

lm = dspy.OpenAI(model="gpt-4o-mini", model_type="chat")
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(rm=colbertv2, lm=lm)

class BasicMultiHop(dspy.Module):
  def __init__(self, passages_per_hop=3):
    self.retrieve = dspy.Retrieve(k=passages_per_hop)
    self.generate_query = dspy.ChainOfThought("context, question -> search_query")
    self.generate_answer = dspy.ChainOfThought("context, question -> answer")

  def forward(self, question):
    context = []

    for hop in range(2):
      query = self.generate_query(context=context, question=question)
      context += self.retrieve(query.search_query).passages

    # this was a single line: "return self.generate_answer(context=context, question=question)" 
    # spliting to two lines to allow tracing to capture the intermediate value, however there is a fundamental problem here ..
    # but works for now
    answer = self.generate_answer(context=context, question=question) 
    return answer


# dummy model that responds with the last 3 words of the prompt and a random query
def finetuned_llm(prompt):
    random_queries = [
        "Who is the king of Saudi Arabia?",
        "How many people live in Spain?",
        "What's the riemann hypothesis?",
        "Who is the president of the United States?",
        "What is the population of China?",
        "What is the largest mammal?",
        "What is the largest planet in the solar system?",
        "What is the capital of Australia?",
        "What is the capital of Japan?",
    ]
    return [f"Last 3 words before this are: [{', '.join(prompt.split()[-3:])}]. now, {random.choice(random_queries)}"]


if __name__ == "__main__":
  # https://github.com/stanfordnlp/dspy/blob/main/examples/qa/hotpot/multihop_finetune.ipynb
  dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=1000, test_size=0)
  trainset = [x.with_inputs('question') for x in dataset.train]
  devset = [x.with_inputs('question') for x in dataset.dev]
  testset = [x.with_inputs('question') for x in dataset.test]

  prog = BasicMultiHop()

  training_data_size = 1
  inference_data_size = 1

  # generate training samples
  dataset_path = 'dataset/train'
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

  for i in range(training_data_size):
    inp = dict()
    inp['question'] = trainset[i].question
    sample = get_traced_sample(prog.forward, inp)
    with open(f'{dataset_path}/{i}.txt', 'w') as f:
      f.write(sample)

  # use model for inference mode
  dataset_path = 'dataset/inference'
  if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

  for i in range(inference_data_size):
    inp = dict()
    inp['question'] = devset[i].question
    sample = get_traced_sample(program=prog.forward, program_inputs=inp, finetuned_model=finetuned_llm)
    with open(f'{dataset_path}/{i}.txt', 'w') as f:
      f.write(sample)
