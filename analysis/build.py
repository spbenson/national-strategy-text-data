from datetime import datetime, timezone

from .data import import_coded_data, import_uncoded_data, prepare_data_train, prepare_data_simple, load_prediction_data
from .llm_load_train import fine_tune_train, fine_tune_test, zero_shot_test
from .transformer_load_train import transformer_train, transformer_test
from .evaluate import evaluate
from .utils import free_gpu_memory
from .predict import llm_predict, transformer_predict, calibrate


def train_test_models(data_path, models_path, results_path,
                      transformer_model_sources,
                      llm_prompt_types, llm_model_sources,
                      get_llm_untrained_results=True, get_llm_trained_results=False
                      transformer_use_class_weights=True, llm_use_class_weights=False,
                      transformer_num_epochs=3, transformer_lr=2e-5,
                      llm_num_epochs=3, llm_lr=1e-4, batch_size=10):

    dtg = datetime.now(timezone.utc).strftime('%d%H%M%Z%y')
    import_coded_data(data_path)
    data_source = data_path + "/coded_natsec.csv"
    results_path_full = results_path + f"/{dtg}_results.txt"
    
    # transformer models first
    for model_source in transformer_model_sources:
        print(f"Running with {model_source}")
        train_dataloader, eval_dataloader, test_dataloader, test_labels = prepare_data_train(
            data_source, prompt_type="encoder", model_source=model_source, is_encoder_model=True, batch_size=batch_size)

        model = transformer_train(
            train_dataloader, eval_dataloader, model_source,
            models_path + f"/trained_{model_source}",
            num_epochs=transformer_num_epochs, lr=transformer_lr,
            use_class_weights=transformer_use_class_weights,
        )
        preds = transformer_test(model, test_dataloader)
        evaluate(test_labels, preds, model_source, results_path_full)
        free_gpu_memory(model, train_dataloader, eval_dataloader, test_dataloader)

    if llm_model_sources:
        for model_source in llm_model_sources:
            print(f"Running with {model_source}")
            for prompt_type in llm_prompt_types:
                print(f"Running with {prompt_type}")

                if get_llm_untrained_results:
                    _, _, x_test = prepare_data_simple(data_source, prompt_type=prompt_type)
                    labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
                    mapping = {label: idx for idx, label in enumerate(labels)}
                    x_test['labels'] = x_test['Label'].map(mapping)
                    zero_shot_preds = zero_shot_test(x_test, model_source)
                    evaluate(x_test['labels'], zero_shot_preds,
                             "ZERO SHOT-" + model_source + "-" + prompt_type, results_path_full)

                if get_llm_trained_results:
                    train_dataloader, eval_dataloader, test_dataloader, test_labels = prepare_data_train(
                        data_source, prompt_type=prompt_type,
                        model_source=model_source, is_encoder_model=False, batch_size=batch_size)

                    model = fine_tune_train(
                        train_dataloader, eval_dataloader, model_source=model_source,
                        output_dir=models_path + f"/trained_{model_source}",
                        num_epochs=llm_num_epochs, lr=llm_lr,
                        use_class_weights=llm_use_class_weights,
                    )
                    fine_tune_preds = fine_tune_test(model, test_dataloader)
                    evaluate(test_labels, fine_tune_preds,
                            "FINE TUNED-" + model_source + "-" + prompt_type, results_path_full)

                    free_gpu_memory(model, train_dataloader, eval_dataloader, test_dataloader)
                    del model, train_dataloader, eval_dataloader, test_dataloader


def train_predict(data_path, results_path, is_transformer,
                  model_source, llm_prompt_type, num_epochs=3, lr=1e-4, batch_size=10):
    dtg = datetime.now(timezone.utc).strftime('%d%H%M%Z%y')
    import_coded_data(data_path)
    import_uncoded_data(data_path)
    coded_data_source = data_path + "/coded_natsec.csv"
    uncoded_data_source = data_path + "/uncoded_natsec.csv"
    results_path_full = results_path + f"/{dtg}_labels.csv"

    if is_transformer:
        train_dataloader, eval_dataloader, _, _ = prepare_data_train(
            coded_data_source, prompt_type="encoder", model_source=model_source, is_encoder_model=True, batch_size=batch_size)
        model = transformer_train(train_dataloader, eval_dataloader, model_source=model_source, num_epochs=num_epochs, lr=lr)

        # Fit temperature on eval set and report ECE before/after
        optimal_temp = calibrate(model, eval_dataloader)

        df, prediction_dataloader = load_prediction_data(uncoded_data_source, prompt_type="encoder",
                         model_source=model_source, batch_size=batch_size,
                         is_encoder_model=True)
        # Run prediction on unlabeled data with calibrated probabilities
        df, preds = transformer_predict(
            model, prediction_dataloader, df,
            output_path=results_path_full,
            temperature=optimal_temp)

    else:
        train_dataloader, eval_dataloader, _, _ = prepare_data_train(
            coded_data_source, prompt_type=llm_prompt_type,
            model_source=model_source, is_encoder_model=False, batch_size=batch_size)
        model = fine_tune_train(train_dataloader, eval_dataloader, model_source=model_source, num_epochs=num_epochs, lr=lr)

        # Fit temperature on eval set and report ECE before/after
        optimal_temp = calibrate(model, eval_dataloader)

        df, prediction_dataloader = load_prediction_data(uncoded_data_source, prompt_type=llm_prompt_type,
                         model_source=model_source, batch_size=batch_size,
                         is_encoder_model=False)
        # Run prediction on unlabeled data with calibrated probabilities
        df, preds = llm_predict(
            model, prediction_dataloader, df,
            output_path=results_path_full,
            temperature=optimal_temp)

    return preds

