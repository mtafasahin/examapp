
import { Component, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterModule, RouterOutlet } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import {MatSidenavModule} from '@angular/material/sidenav';
import { MatMenuModule } from '@angular/material/menu';
import { AuthService } from '../../services/auth.service';
import { SidenavService } from '../../services/sidenav.service';

@Component({
  selector: 'app-layout',
  imports: [CommonModule,
    MatSidenavModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    MatListModule,
    RouterModule,
    MatMenuModule
  ],
  templateUrl: './layout.component.html',
  styleUrl: './layout.component.scss'
})
export class LayoutComponent {
  // isSidenavOpen = signal(true);
  showHeader = signal(true);

  authService = inject(AuthService);
  profileImage = 'assets/profile.png';  
  userRole: string | null = null;
  $isAuthenticated = this.authService.isAuthenticated();
  sidenavService = inject(SidenavService);
  toggleSidenav() {
    this.sidenavService.toggleSidenav();
  }

  toggleFullScreen() {
    this.sidenavService.toggleFullScreen();
  }

  isLoginPage(): boolean {
    return this.router.url === '/login';
  }

  
  isSidenavOpen = this.sidenavService.isSidenavOpen; // Servisten değer alıyoruz
  isFullScreen = this.sidenavService.isFullScreen; // Servisten değer alıyoruz

  constructor(private router: Router) {
    this.router.events.subscribe(event => {
      if (event instanceof NavigationEnd) {
        const url = (event as NavigationEnd).urlAfterRedirects; // ✅ Doğru URL almak için
        this.showHeader.set(!(url === '/login' || url === '/register'));
      }
    });
  }
  title = 'exam-app';
    get userAvatarUrl () {
      return this.authService.getUserAvatar() || this.profileImage;
    }
  
    navigateTo(path: string) {
      this.router.navigate([path]);
    }
  
    logout() {
      this.authService.logout();
    }
}

