# -*- coding: utf-8 -*-
from django.conf.urls import patterns, url
from myproject.myapp import views as my_view

urlpatterns = patterns('myproject.myapp.views',
    url(r'^home/$', my_view.home, name='home'),
    url(r'^load/$', my_view.load, name='load'),
    url(r'^dosomething/$', my_view.dosomething, name='dosomething'),
    url(r'^exploration/$', my_view.exploration, name='exploration'),
    url(r'^DIY/$', my_view.DIY, name='DIY'),
    url(r'^tutorial/$', my_view.tutorial, name='tutorial'),
    url(r'^about/$', my_view.about, name='about'),
    url(r'^contact/$', my_view.contact, name='contact'),
)

