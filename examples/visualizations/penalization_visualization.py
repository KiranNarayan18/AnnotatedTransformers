import torch
import altair as alt
import pandas as pd

from models.config import learning_rate

from models.regularization import LabelSmoothing

def loss(x, crit):
  d = x + 3 * 1
  predict = torch.tensor([[0, x/d, 1/d, 1/d, 1/d]], dtype=torch.float32)
  return crit(torch.log(predict), torch.tensor([1], dtype=torch.long)).data

def penalization_visualizatoin():
    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.1)
    loss_data = pd.DataFrame(
                        {
                            "Loss": [loss(x, crit) for x in range(1, 100)],
                            "Steps": list(range(99))
                        }
                    ).astype("float")

    chart =  (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=600)
        .encode(
            x="Steps",
            y="Loss",
        ).interactive()
    )

    chart.show()

    return chart






alt.renderers.enable("browser")

RUN_EXAMPLES = True
def show_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    return fn(*args)


show_example(penalization_visualizatoin)