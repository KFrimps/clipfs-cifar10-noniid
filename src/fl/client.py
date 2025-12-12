class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cfg, client_idx):
        self.model = model.to(cfg.device)
        self.trainloader = train_loader
        self.testloader = test_loader
        self.cfg = cfg
        self.client_idx = client_idx 
        
        # Define the Loss Function
        # (You can keep label_smoothing if you want the accuracy boost)
        self.crit = nn.CrossEntropyLoss() 

        # Define the Optimizer ONCE using the fixed global LR
        self.opt = torch.optim.SGD(
            self.model.parameters(), 
            lr=cfg.lr,                  
            momentum=cfg.momentum, 
            weight_decay=cfg.weight_decay
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1. Update Local Model with Global Weights
        self.set_parameters(parameters)

        # 2. Get the Epoch Count
        epochs = int(config.get("local_epochs", self.cfg.local_epochs))
        
        # 3. Standard Training Loop 
        self.model.train()
        
        for epoch in range(epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.cfg.device, non_blocking=True), y.to(self.cfg.device, non_blocking=True)
                
                # Type casting 
                if x.dtype != torch.float32: x = x.float()
                if y.dtype != torch.long: y = y.long()

                self.opt.zero_grad()
                logits = self.model(x)
                loss = self.crit(logits, y)
                loss.backward()
                self.opt.step()
        
        # 4. Return Updates
        # We simply return the fixed values in metrics for logging consistency
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "cv_epochs": epochs,       # Log the fixed number used
            "cv_lr": self.cfg.lr,      # Log the fixed LR used
            "client_idx": self.client_idx
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                x = x.float()
                logits = self.model(x)
                loss_sum += self.crit(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        
        loss = float(loss_sum/total) if total > 0 else 0.0
        acc = float(correct/total) if total > 0 else 0.0

        return loss, total, {
            "accuracy": acc,
            "loss": loss,
            "client_idx": self.client_idx
        }
