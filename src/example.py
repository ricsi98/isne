import torch
from lightning import Trainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.datasets import Planetoid

from loader import BatchedRandomWalkLoaderBuilder
from model import ISNE


def main():
    torch.set_float32_matmul_precision("high")

    dataset = Planetoid(root="data", name="Cora")
    graph = dataset[0]

    loader = BatchedRandomWalkLoaderBuilder(
        edge_index=graph.edge_index, walk_length=10, context_size=5, walks_per_node=10, num_negative_samples=1
    ).build()

    model = ISNE(num_nodes=graph.num_nodes, hidden_channels=16, edge_index=graph.edge_index)

    trainer = Trainer(max_epochs=10)
    trainer.fit(model, loader)

    # Evaluation
    embeddings = model.embed_nodes(graph.edge_index)
    labels = graph.y
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, labels, stratify=labels, test_size=0.1, random_state=42
    )
    knn = KNeighborsClassifier(15, metric="cosine")
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
