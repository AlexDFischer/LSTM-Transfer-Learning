#!/bin/python

def LSTMCell(input, hidden, ws, bs=None):
    if input.is_cuda:
        assert False
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused()
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    num_layers = len(w_ihs)
    if bs != None and len(bs) != num_layers:
      throw "length of ws (%d) needs to equal length of bs (%d)" % (len(ws), len(bs))
    
    input_and_hidden = torch.cat(input, hx)
    gates = input_and_hidden
    for layer in range(num_layers - 1):
      gates = F.linear(gates, ws[layer], bs[layer])
      gates = F.tanh(gates)
    gates = F.linear(gates, ws[-1], bs[-1])
    #gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy
