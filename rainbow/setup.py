from google.colab import drive
drive.mount("/content/drive")

import os
os.environ['PYTHONPATH'] += ':/content/drive/My Drive/flatland-rl/'
! echo $PYTHONPATH

!pip install flatland-rl
!pip install torch_geometric
!pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html



function ClickConnect() {
  console.log('Working')
  document
    .querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect')
    .click()
}

setInterval(ClickConnect, 60000)