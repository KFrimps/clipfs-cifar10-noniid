def make_metric_logger(out_csv):
    # Turn string into a Path object
    out_csv = Path(out_csv)

    # Make sure the parent directory exists (runs/IID)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist yet, create it with headers
    if not out_csv.exists():
        pd.DataFrame(columns=["round", "accuracy", "loss"]).to_csv(out_csv, index=False)

    def aggregate_and_log(metrics):
        # metrics = list of (num_examples, {"accuracy": acc, "loss": loss})
        acc_sum, loss_sum, n_sum = 0.0, 0.0, 0
        for num_examples, m in metrics:
            acc_sum  += num_examples * m["accuracy"]
            loss_sum += num_examples * m["loss"]
            n_sum    += num_examples

        acc  = acc_sum / n_sum
        loss = loss_sum / n_sum

        # current round = number of existing rows + 1
        current_round = len(pd.read_csv(out_csv)) + 1

        # append new row
        pd.DataFrame([{
            "round": current_round,
            "accuracy": acc,
            "loss": loss
        }]).to_csv(out_csv, mode="a", header=False, index=False)

        print(f"[Round {current_round:02d}] GLOBAL acc={acc:.4f}, loss={loss:.4f}")
        return {"accuracy": acc, "loss": loss}

    return aggregate_and_log
