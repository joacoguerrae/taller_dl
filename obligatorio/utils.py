import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


def evaluate(model, criterion, data_loader, device):
    """
    Evalúa el modelo en los datos proporcionados y calcula la pérdida promedio.

    Args:
        model (torch.nn.Module): El modelo que se va a evaluar.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        data_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de evaluación.

    Returns:
        float: La pérdida promedio en el conjunto de datos de evaluación.

    """
    model.eval()  # ponemos el modelo en modo de evaluacion
    total_loss = 0  # acumulador de la perdida
    total_dice = 0  # acumulador del dice score

    # Mixed precision para evaluación también
    from torch.cuda.amp import autocast

    use_amp = torch.cuda.is_available()

    with torch.no_grad():  # deshabilitamos el calculo de gradientes
        for x, y in data_loader:  # iteramos sobre el dataloader
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo

            if use_amp:
                with autocast():
                    output = model(x)  # forward pass
                    total_loss += criterion(output, y).item()  # acumulamos la perdida
                    total_dice += dice_score(
                        output, y
                    ).item()  # acumulamos el dice score
            else:
                output = model(x)
                total_loss += criterion(output, y).item()
                total_dice += dice_score(output, y).item()

            # Limpieza de memoria
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / len(data_loader), total_dice / len(
        data_loader
    )  # retornamos la perdida promedio y el dice score promedio


class EarlyStopping:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(epoch, train_loss, val_loss, val_dice):
    print(
        f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Dice: {val_dice:.5f}"
    )


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    do_early_stopping=True,
    patience=5,
    epochs=10,
    log_fn=print_log,
    log_every=1,
    checkpoint_dir=None,
    save_every=10,
):
    """
    Entrena el modelo utilizando el optimizador y la función de pérdida proporcionados.

    Args:
        model (torch.nn.Module): El modelo que se va a entrenar.
        optimizer (torch.optim.Optimizer): El optimizador que se utilizará para actualizar los pesos del modelo.
        criterion (torch.nn.Module): La función de pérdida que se utilizará para calcular la pérdida.
        train_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader que proporciona los datos de validación.
        device (str): El dispositivo donde se ejecutará el entrenamiento.
        patience (int): Número de épocas a esperar después de la última mejora en val_loss antes de detener el entrenamiento (default: 5).
        epochs (int): Número de épocas de entrenamiento (default: 10).
        log_fn (function): Función que se llamará después de cada log_every épocas con los argumentos (epoch, train_loss, val_loss) (default: None).
        log_every (int): Número de épocas entre cada llamada a log_fn (default: 1).

    Returns:
        Tuple[List[float], List[float]]: Una tupla con dos listas, la primera con el error de entrenamiento de cada época y la segunda con el error de validación de cada época.

    """

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_train_errors = []  # colectamos el error de traing para posterior analisis
    epoch_val_errors = []  # colectamos el error de validacion para posterior analisis
    epoch_dice_scores = []  # colectamos el dice score de validacion para posterior analisis

    # Mixed Precision Training - REDUCE MEMORIA EN 50%
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()
    use_amp = torch.cuda.is_available()

    if do_early_stopping:
        early_stopping = EarlyStopping(
            patience=patience
        )  # instanciamos el early stopping

    for epoch in range(epochs):  # loop de entrenamiento
        model.train()  # ponemos el modelo en modo de entrenamiento
        train_loss = 0  # acumulador de la perdida de entrenamiento
        for x, y in train_loader:
            x = x.to(device)  # movemos los datos al dispositivo
            y = y.to(device)  # movemos los datos al dispositivo

            optimizer.zero_grad()  # reseteamos los gradientes

            # Forward pass con mixed precision
            if use_amp:
                with autocast():
                    output = model(x)  # forward pass (prediccion)
                    batch_loss = criterion(output, y)  # calculamos la perdida

                # Backward con scaler
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                batch_loss = criterion(output, y)
                batch_loss.backward()
                optimizer.step()

            train_loss += batch_loss.item()  # acumulamos la perdida

            # Limpieza de memoria después de cada batch
            del output, batch_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)  # calculamos la perdida promedio de la epoca
        epoch_train_errors.append(train_loss)  # guardamos la perdida de entrenamiento

        val_loss, val_dice = evaluate(
            model, criterion, val_loader, device
        )  # evaluamos el modelo en el conjunto de validacion
        epoch_val_errors.append(val_loss)  # guardamos la perdida de validacion
        epoch_dice_scores.append(val_dice)  # guardamos el dice score de validacion

        if checkpoint_dir is not None and ((epoch + 1) % save_every == 0):
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

        if do_early_stopping:
            early_stopping(val_loss)  # llamamos al early stopping
            # si la perdida de validacion mejoro, guardamos el modelo en una variable temporal
            if early_stopping.counter == 0 and checkpoint_dir is not None:
                best_model_weights = model.state_dict().copy()
                
                

        if log_fn is not None:  # si se pasa una funcion de log
            if (epoch + 1) % log_every == 0:  # loggeamos cada log_every epocas
                log_fn(
                    epoch, train_loss, val_loss, val_dice
                )  # llamamos a la funcion de log

        if do_early_stopping and early_stopping.early_stop:
            # cargamos los mejores pesos guardados y guardamos los pesos en checkpoint_dir
            if checkpoint_dir is not None:
                model.load_state_dict(best_model_weights)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Detener entrenamiento en la época {epoch}, la mejor pérdida fue {early_stopping.best_score:.5f}"
            )
            break

    return epoch_train_errors, epoch_val_errors, epoch_dice_scores


def dice_score(logits, targets, eps=1e-6):
    """
    logits: salida del modelo (B, 1, H, W), sin sigmoid
    targets: máscara ground truth (B, 1, H, W), 0/1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    targets = targets.float()
    # sumamos por batch y espacio
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def plot_taining(train_errors, val_errors):
    # Graficar los errores
    plt.figure(figsize=(10, 5))  # Define el tamaño de la figura
    plt.plot(train_errors, label="Train Loss")  # Grafica la pérdida de entrenamiento
    plt.plot(val_errors, label="Validation Loss")  # Grafica la pérdida de validación
    plt.title("Training and Validation Loss")  # Título del gráfico
    plt.xlabel("Epochs")  # Etiqueta del eje X
    plt.ylabel("Loss")  # Etiqueta del eje Y
    plt.legend()  # Añade una leyenda
    plt.grid(True)  # Añade una cuadrícula para facilitar la visualización
    plt.show()  # Muestra el gráfico


def model_calassification_report(model, dataloader, device, nclasses):
    # Evaluación del modelo
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcular precisión (accuracy)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Reporte de clasificación
    report = classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(nclasses)]
    )
    print("Reporte de clasificación:\n", report)


def show_tensor_image(tensor, title=None, vmin=None, vmax=None):
    """
    Muestra una imagen representada como un tensor.

    Args:
        tensor (torch.Tensor): Tensor que representa la imagen. Size puede ser (C, H, W).
        title (str, optional): Título de la imagen. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    # Check if the tensor is a grayscale image
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:  # Assume RGB
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(tensors, titles=None, figsize=(15, 5), vmin=None, vmax=None):
    """
    Muestra una lista de imágenes representadas como tensores.

    Args:
        tensors (list): Lista de tensores que representan las imágenes. El tamaño de cada tensor puede ser (C, H, W).
        titles (list, optional): Lista de títulos para las imágenes. Por defecto es None.
        vmin (float, optional): Valor mínimo para la escala de colores. Por defecto es None.
        vmax (float, optional): Valor máximo para la escala de colores. Por defecto es None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        # Check if the tensor is a grayscale image
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:  # Assume RGB
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()
