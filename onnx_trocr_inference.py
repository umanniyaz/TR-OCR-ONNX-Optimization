
import os
import time
from typing import Optional, Tuple

import torch
from PIL import Image

import onnxruntime as onnxrt
import requests
from transformers import AutoConfig, AutoModelForVision2Seq, TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


device = torch.device("cpu")
session_options = onnxrt.SessionOptions()
session_options.intra_op_num_threads = 12
session_options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL


model_name = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

url = r"<path>"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open(url).convert("RGB")
pixel_values = processor([image], return_tensors="pt").pixel_values

class ORTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_input_name = "pixel_values"
        self._device = device     
        self.session = onnxrt.InferenceSession(
            os.path.join(os.getcwd(),'models_trocr_base\\encoder_model.onnx'), providers=["CPUExecutionProvider"], sess_options=session_options
        )
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutput:

        onnx_inputs = {"pixel_values": pixel_values.cpu().detach().numpy()}

        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]]).to(self._device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = device

        self.session = onnxrt.InferenceSession(
            os.path.join(os.getcwd(),'models_trocr_base\\decoder_model.onnx'), providers=["CPUExecutionProvider"],sess_options=session_options
        )

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> Seq2SeqLMOutput:

        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
        }

        if "attention_mask" in self.input_names:
            onnx_inputs["attention_mask"] = attention_mask.cpu().detach().numpy()

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names:
            onnx_inputs["encoder_hidden_states"] = encoder_hidden_states.cpu().detach().numpy()

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self._device)
        return Seq2SeqLMOutput(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
        }


class ORTModelForVision2Seq(VisionEncoderDecoderModel, GenerationMixin):
    def __init__(self, *args, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)
        self._device = device

        self.encoder = ORTEncoder()
        self.decoder = ORTDecoder()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(pixel_values=pixel_values.to(device))

        # Decode
        decoder_attention_mask = decoder_input_ids.new_ones(decoder_input_ids.shape)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):

        return {
            "decoder_input_ids": input_ids,
            "decoder_atttention_mask": input_ids,
            "encoder_outputs": encoder_outputs,
        }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value

    def to(self, device):
        self.device = device
        return self


def test_ort():
    model = ORTModelForVision2Seq()
    model = model.to(device)

    start = time.time()

    model.config.decoder_start_token_id = 2
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.pad_token_id = model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = model.config.decoder.eos_token_id = processor.tokenizer.sep_token_id

    generated_ids = model.generate(pixel_values.to(device))

    end = time.time()

    model_output = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, device=device)[0]
    print("Model Ouput ORT is : ",model_output)

    print("ORT time: ", end - start, model_output)


def test_original():
    start = time.time()
    generated_ids = model.generate(pixel_values.to(device))
    end = time.time()

    model_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Model Output Original is : ",model_output)

    print("Original time: ", end - start, model_output)


test_original()
test_ort()








