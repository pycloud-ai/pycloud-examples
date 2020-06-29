#
# Copyright (C) 2020 PyCloud - All Rights Reserved
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from pycloud.core import PyCloud

os.environ["TORCH_HOME"] = "./cache"
os.environ["PYTORCH_FAIRSEQ_CACHE"] = "./cache"
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

import torch  # noqa
import fairseq  # noqa


CLOUD = PyCloud.get_instance()


@CLOUD.init_service(service_id="translator")
def init_model():
    # Load a transformer trained on WMT'16 En-De
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses',
                           bpe='subword_nmt')
    # en2de.cuda()
    en2de.eval()  # disable dropout

    # The underlying model is available under the *models* attribute
    assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)
    return {'model': en2de}


@CLOUD.endpoint(service_id="translator")
def translate(text):
    return CLOUD.initialized_data()['model'].translate(text)


if __name__ == "__main__":
    init_model()

    translate('Hello world!')
    translate(['Hello world!', 'The cat sat on the mat.'])

    CLOUD.configure_service("translator", exposed=True, package_deps=["gcc", "g++"])
    CLOUD.set_basic_auth_credentials("pycloud", "demo")
    CLOUD.save()
