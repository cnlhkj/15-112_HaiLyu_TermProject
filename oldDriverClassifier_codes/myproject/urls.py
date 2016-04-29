from django.conf.urls import include, url
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

from django.contrib import admin

import registration.backends.default

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^myapp/',include('myproject.myapp.urls')),
    url(r'^accounts/', include('registration.backends.simple.urls')),
    url(r'^$', RedirectView.as_view(url='/myapp/home/', permanent=True)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
