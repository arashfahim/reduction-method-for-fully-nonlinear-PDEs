import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def Grad(x,v): #output= [M,D,D], #input: x=[M,D], t=[M,1], xt= [M,D+1]
    d = x.shape[1]
    Du=torch.zeros(x.shape[0],d)#.to(device)
    xin=x.clone().detach()
    xin.requires_grad=True
    u=v(xin)
    Du=torch.autograd.grad(outputs=[u],inputs=[xin],grad_outputs=torch.ones_like(u),
                           allow_unused=True,retain_graph=True,create_graph=True)[0].unsqueeze(2)
    Du = torch.reshape(Du,(Du.shape[0],d,1))
    return Du

def Grad_Hess(x,v): #output= [M,D,D], #input: x=[M,D], t=[M,1], xt= [M,D+1]
    d = x.shape[1]
    hess_temp=torch.zeros(x.shape[0],d,d)#.to(device)
    Du=torch.zeros(x.shape[0],d)#.to(device)
    xin=x.clone().detach()
    xin.requires_grad=True
    u=v(xin)
    Du=torch.autograd.grad(outputs=[u],inputs=[xin],grad_outputs=torch.ones_like(u),
                           allow_unused=True,retain_graph=True,create_graph=True)[0].unsqueeze(2)
    hess_temp= torch.cat([torch.autograd.grad(outputs=[Du[:,i,:]],inputs=[xin],grad_outputs=torch.ones_like(Du[:,i,:]),
                           allow_unused=True,retain_graph=True,create_graph=True)[0] for i in range(d)],1)
    Du = torch.reshape(Du,(Du.shape[0],d,1))
    hess_temp=torch.reshape(hess_temp,(hess_temp.shape[0],d,d))
    return Du, hess_temp