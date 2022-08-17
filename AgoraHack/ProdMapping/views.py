from django.views.generic import ListView
from django.http import HttpResponse

def hey(request):
    f = 'hey now'
    return HttpResponse(f)