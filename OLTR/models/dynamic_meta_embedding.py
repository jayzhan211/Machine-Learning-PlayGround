import torch
import torch.nn as nn
from models.cosine_norm_classifer import Cosine_Norm_Classifier

class Meta_Embedding_Classifier(nn.Module):
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(Meta_Embedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = Cosine_Norm_Classifier(feat_dim, num_classes)

    def forward(self, x, centroids, *args):

        direct_feature = x
        batch_size, feat_size = x.size(0), x.size(1)

        ### Visual Memory
        x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids

        dist_cur = torch.norm(x_expand - centroids_expand, p='fro', dim=2)
        # size: dis_cur = (batch_size, num_classes)
        values_nn = torch.min(dist_cur, dim=1)[0]
        # size: values_nn = (batch_size)
        scale = 10.0
        reachability = (scale / values_nn).unsqueeze(1).expand(-1, feat_size)

        values_memory = self.fc_hallucinator(x)
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        concept_selector = self.fc_selector(x)
        concept_selector = concept_selector.tanh()
        x = reachability * (direct_feature + concept_selector * memory_feature)

        infused_feature = concept_selector * memory_feature

        logits = self.cosnorm_classifier(x)
        return logits, [direct_feature, infused_feature]




        
