import unittest
import torch
import sys


class TestFederatedTraining(unittest.TestCase):

    def test_same_weights_net(self, sent_net, received_net):
        #print("Check if server well sent weights to users")
        for layer in sent_net.state_dict():
            assert torch.all(torch.eq(sent_net.state_dict()[layer], received_net.state_dict()[layer])), 'Weights in layer {} not equal'.format(layer)

    def test_well_received(self, server, user, nk):
        #print("Check if server well received users' weights")
        for layer in user.net.state_dict():
            #print(torch.sum(server.received_weights[layer][-1]))
            #print(torch.sum(nk * user.net.state_dict()[layer]))
            if not torch.all(torch.eq(server.received_weights[layer][-1],
                                      nk*user.net.state_dict()[layer])):
                #print(layer)
                sys.exit("Server didn't receive correct weights")

    def test_well_updated_server(self, server,N):
        #print("Check if server have well updated weights")
        for layer in server.received_weights:
            if not torch.all(torch.eq(server.net.state_dict()[layer], N *
                                      torch.sum(torch.stack(server.received_weights[layer]), 0, keepdim=True).squeeze(
                                          0))):
                sys.exit("Server didn't update weights correctly")
