'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-28 11:55:34
LastEditors: Andy
LastEditTime: 2022-01-28 12:18:52
'''

import sys
from typing import List

from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models



class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
        access_key_id: str,
        access_key_secret: str,
    ) -> Dysmsapi20170525Client:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 您的AccessKey ID,
            access_key_id="LTAI5tC8gZSPVqxYs71LX5mQ",
            # 您的AccessKey Secret,
            access_key_secret="Voqzn7x1MUMofmYkbKUvGD45KvPcnj"
        )
        # 访问的域名
        config.endpoint = f'dysmsapi.aliyuncs.com   '
        return Dysmsapi20170525Client(config)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client('accessKeyId', 'accessKeySecret')
        dysmsapi_20170525_models.SendSmsRequest(
            PhoneNumbers="15057271937",
            SignName=
        )
        add_short_url_request = dysmsapi_20170525_models.AddShortUrlRequest(
            resource_owner_account='your_value',
            resource_owner_id=1,
            source_url='your_value',
            short_url_name='your_value'
        )
        # 复制代码运行请自行打印 API 的返回值
        client.add_short_url(add_short_url_request)

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client('accessKeyId', 'accessKeySecret')
        add_short_url_request = dysmsapi_20170525_models.AddShortUrlRequest(
            resource_owner_account='your_value',
            resource_owner_id=1,
            source_url='your_value',
            short_url_name='your_value'
        )
        # 复制代码运行请自行打印 API 的返回值
        await client.add_short_url_async(add_short_url_request)


if __name__ == '__main__':
    Sample.main("hello")
