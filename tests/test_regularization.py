from models.regularization import LabelSmoothing
import torch
import altair as alt
import pandas as pd


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.tensor(
      [
          [0, 0.2, 0.7, 0.1, 0],
          [0, 0.2, 0.7, 0.1, 0],
          [0, 0.2, 0.7, 0.1, 0],
          [0, 0.2, 0.7, 0.1, 0],
          [0, 0.2, 0.7, 0.1, 0],
      ],
      dtype=torch.float32,
    )

    crit(x=torch.log(predict), target=torch.tensor([2, 1, 0, 3, 3], dtype=torch.long))


    LS_data = pd.concat([
        pd.DataFrame(
            {
                "target distribution": crit.true_dist[x,y].flatten(),
                "columns": y,
                "rows": x
            }) 
                for y in range(5)
                for x in range(5)
    ])

    chart = (alt.Chart(LS_data)
            .mark_rect(color="Blue", opacity=1)
            .properties(height=400, width=400)
            .encode(
                alt.X("columns:O", title=None),
                    alt.Y("rows:O", title=None),
                    alt.Color(
                        "target distribution:Q", scale=alt.Scale(scheme="viridis")
                    ),
                ).interactive())
    
    chart.show()

    return chart



alt.renderers.enable("browser")

RUN_EXAMPLES = True
def show_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    return fn(*args)


show_example(example_label_smoothing)