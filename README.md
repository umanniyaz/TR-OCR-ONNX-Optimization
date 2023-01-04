# TR-OCR-ONNX-Optimization

For Tr-OCR model conversion to ONNX will give two file encoder_onnx and decoder_onnx,use low value for tolerance `--atol` 1e-3:

`python -m transformers.onnx --model=microsoft/trocr-base-printed  --feature=vision2seq-lm models_trocr_base --atol 1e-3`


FOR CPU as Execution provider,we have sequential vs parallel execution

FOR SEQUENTIAL:
 `sess_options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL`
 
Thread Count:
 `sess_options.intra_op_num_threads = 2` (controls the number of threads to use to run the model )

FOR PARALLEL EXECUTION:
`sess_options.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL`
`sess_options.inter_op_num_threads = 2` (to control the number of threads used to parallelize the execution of the graph (across nodes)

FOR GPU (CUDA) as Execution provider use iobinding:

Example:
`io_binding = session.io_binding()
io_binding.bind_ortvalue_input('X', x_ortvalue)
io_binding.bind_ortvalue_output('Y', y_ortvalue)
session.run_with_iobinding(io_binding)`

Try adjusting `inter_op_threads` and `intra_op_threads` for Sequential and Parallel Execution(CPU) and iobinding on CUDA (GPU), I believe you will see improvement in Inference with actual model accuracy preserved

Refernces:
https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Graph_Optimizations.md
https://fs-eire.github.io/onnxruntime/docs/performance/tune-performance.html


