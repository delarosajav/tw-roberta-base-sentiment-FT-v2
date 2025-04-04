import gradio as gr
from transformers import pipeline

pipe = pipeline(model="delarosajav95/tw-roberta-base-sentiment-FT-v2")
#function that Gradio will use to classify
def classify_text(inputs):
  result = pipe(inputs, return_all_scores=True)
  output = []
  label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
  for i, predictions in enumerate(result):
    for pred in predictions:
      label = label_mapping.get(pred['label'], pred['label'])
      score = pred['score']
      output.append(f"{label}: {score:.2%}")

  return "\n".join(output)
#defining Gradio interface
textbox = gr.Textbox(lines=3, placeholder="Enter a user review, comment, or opinion to evaluate...(e.g., 'I love this product! It looks great.')",
                     label="User Review/Comment:")

output_box = gr.Textbox(label="Results:")

iface = gr.Interface(
    fn=classify_text, 
    inputs=textbox,
    outputs=output_box,
    live=True, 
    title="Sentiment Analysis for User Opinions & Feedback",
    allow_flagging="never",
)

# Launch the interface
iface.launch()