from sklearn import metrics

def evaluate(y_true_mapped, y_pred_mapped, model_name, results_file_path):
    """
    Takes model predictions and prints performance + writes it to file.

    y_true: true labels
    y_pred: predicted labels
    """
    print(f"Printing Model Results for {model_name}:")
    with open(results_file_path, "a", encoding='utf-8') as results_file:
        results_file.write(f'\n\n--------------\n{model_name}\n--------------\n')
        labels = ["Not_Aligned", "Aligned", "Neutral/Irrelevant"]
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_true_mapped, y_pred_mapped)
        macro_f1 = metrics.f1_score(y_true_mapped, y_pred_mapped, average="macro")
        print(f"Test accuracy: {accuracy:.4f} | Test macro-F1: {macro_f1:.4f}")
        results_file.write(f"Test accuracy: {accuracy:.4f}\nTest macro-F1: {macro_f1:.4f}\n")

        # Generate classification report
        class_report = metrics.classification_report(
            y_true=y_true_mapped, y_pred=y_pred_mapped,
            target_names=labels, labels=list(range(len(labels))))
        print(f'\nClassification Report:\n{class_report}')
        results_file.write(f'\nClassification Report:\n{class_report}\n')

        # Generate confusion matrix
        try:
            conf_matrix = metrics.confusion_matrix(
                y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels))))
            print(f'\nConfusion Matrix:\n{conf_matrix}')
            results_file.write(f'\nConfusion Matrix:\n{conf_matrix}\n')
        except (TypeError, ValueError) as e:
            print(f"{e}: confusion matrix not found")