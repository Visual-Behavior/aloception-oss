import alonet
from alonet.raft.update import SmallUpdateBlock
from alonet.raft.extractor import SmallEncoder

from alonet.raft.raft import RAFTBase


class RAFTSmall(RAFTBase):

    hidden_dim = 96
    context_dim = 64
    corr_levels = 4
    corr_radius = 3

    def __init__(self, dropout=0, **kwargs):
        self.dropout = dropout
        fnet = self.build_fnet(encoder_cls=SmallEncoder, output_dim=128)
        cnet = self.build_cnet(encoder_cls=SmallEncoder)
        update_block = self.build_update_block(update_cls=SmallUpdateBlock)

        super().__init__(fnet, cnet, update_block, **kwargs)


if __name__ == "__main__":
    from torch.utils.data import SequentialSampler
    from alodataset import ChairsSDHomDataset, Split
    from aloscene import Frame

    print()
    print("[Warning] No pretrained weights for RAFTSmall. In this demo, the model is randomly initialized.")
    print()
    raft = RAFTSmall()
    chairs = ChairsSDHomDataset(split=Split.VAL)
    loader = chairs.train_loader(sampler=SequentialSampler)
    frames = next(iter(loader))
    frames = Frame.batch_list(frames)
    frames = frames.norm_minmax_sym()
    frame1 = frames[:, 0, ...]
    frame2 = frames[:, 1, ...]
    model_out = raft.forward(frame1, frame2)
    flows = raft.inference(model_out)
    flow_final = flows[-1].detach().cpu()
    flow_final.get_view().render()
