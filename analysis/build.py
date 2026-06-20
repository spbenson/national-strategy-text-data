from datetime import datetime, timezone

from .data import import_coded_data, import_uncoded_data, prepare_data_train, prepare_data_simple
from .llm_load_train import fine_tune_train, fine_tune_test, zero_shot_test
from .transformer_load_train import transformer_train, transformer_test
from .evaluate import evaluate
from .constants import LABEL_MAPPING

def train_test_models(data_path, models_path, results_path, 
                      transformer_model_sources,
                      llm_prompt_types, llm_model_sources, 
                      get_untrained_results = True):

    dtg = datetime.now(timezone.utc).strftime('%d%H%M%Z%y')
    import_coded_data(data_path)
    data_source = data_path + "/coded_natsec.csv"
    results_path_full = results_path + f"/{dtg}_results.txt"

    for model_source in transformer_model_sources:
        print(f"Running with {model_source}")
        train_dataloader, eval_dataloader, test_dataloader, test_labels = prepare_data_train(
            data_source, prompt_type="encoder", model_source=model_source, is_encoder_model=True)
        model = transformer_train(train_dataloader, eval_dataloader, model_source, models_path + f"/trained_{model_source}")
        preds = transformer_test(model, test_dataloader)
        evaluate(test_labels, preds, model_source, results_path_full)

    for model_source in llm_model_sources:
        print(f"Running with {model_source}")
        for prompt_type in llm_prompt_types:
            print(f"Running with {prompt_type}")

            if get_untrained_results:
                _, _, x_test = prepare_data_simple(data_source, prompt_type=prompt_type)
                x_test['labels'] = x_test['Label'].map(LABEL_MAPPING)
                zero_shot_preds = zero_shot_test(x_test, model_source)
                evaluate(x_test['labels'], zero_shot_preds,
                        "ZERO SHOT-" + model_source, results_path_full)

            train_dataloader, eval_dataloader, test_dataloader, test_labels = prepare_data_train(
                data_path + "/coded_natsec.csv", prompt_type=prompt_type, model_source=model_source, is_encoder_model=False)

            model = fine_tune_train(train_dataloader, eval_dataloader, model_source=model_source, output_dir="drive/MyDrive/LLM_Saves")
            fine_tune_preds = fine_tune_test(model, test_dataloader)

            evaluate(test_labels, fine_tune_preds,
                     "FINE TUNED-" + model_source, results_path_full)
              
def train_predict(data_path, models_path, results_path, 
                      transformer_model_sources,
                      llm_prompt_types, llm_model_sources, 
                      get_untrained_results = True):
    pass
