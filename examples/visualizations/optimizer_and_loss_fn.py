from torch.optim.lr_scheduler import LambdaLR

import torch
import altair as alt
import pandas as pd

from models.config import learning_rate

def example_learning_scheduler():
    opts = [
        [512, 1, 4000],
        [512, 1, 8000],
        [256, 1, 4000]
    ]

    dummy_model = torch.nn.Linear(1,1)
    learning_rates = []

    for idx, example in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9,0.98), eps=1e-9
        )

        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: learning_rate(step, *example)
        )

        tmp = []

        for step in range(20000):
            tmp.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            lr_scheduler.step()

        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, : ],
                    "model_size_warmup": ["512:4000", "512:8000", "256:4000"][warmup_idx],
                    "step": range(20000)
                }
            ) for warmup_idx in [0,1,2]
        ]
    )


    chart = (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x='step', y='Learning Rate', color='model_size_warmup:N')
        .interactive()
    )

    chart.show()

    return chart

alt.renderers.enable("browser")

RUN_EXAMPLES = True
def show_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    return fn(*args)


show_example(example_learning_scheduler)
