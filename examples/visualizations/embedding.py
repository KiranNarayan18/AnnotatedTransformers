
import os
import torch
import pandas as pd
import altair as alt
from models.embeddings import PositionalEncoding


alt.renderers.enable("browser")

def example_positional():
  pe = PositionalEncoding(20, 0)
  y = pe.forward(torch.zeros(1, 100, 20))

  data = pd.concat([
            pd.DataFrame({
            "embedding": y[0, :, dim],
            "dimension": dim,
            "position": list(range(100))
        })  for dim in [4,5,6,7]

      ])


  chart = (
      alt.Chart(data).mark_line().properties(width=800).encode(x="position", y="embedding", color="dimension:N").interactive()
  )

  chart.show()
  return chart
#   chart.save("positional_encoding.html")






