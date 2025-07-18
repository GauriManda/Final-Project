// Header.jsx
import React from 'react';
import { Brain, Home, GraduationCap, Users, Gamepad2 } from 'lucide-react';
import './Header.css';

const Header = ({ 
  currentPage = 'home', 
  onNavigate
}) => {
  const navigationItems = [
    { id: 'home', label: 'Dashboard', icon: Home },
    { id: 'teacher', label: 'Teacher', icon: GraduationCap },
    { id: 'parent', label: 'Parent', icon: Users },
    { id: 'games', label: 'Games', icon: Gamepad2 }
  ];

  const handleNavigation = (item) => {
    if (onNavigate) {
      onNavigate(item);
    }
  };

  return (
    <header className="header">
      <div className="header-content">
        {/* Logo */}
        <div className="logo" onClick={() => handleNavigation(navigationItems[0])}>
          <Brain size={24} />
          <span className="logo-text">MindTracker</span>
        </div>

        {/* Navigation */}
        <nav className="navigation">
          {navigationItems.map((item) => {
            const IconComponent = item.icon;
            const isActive = currentPage === item.id;
            
            return (
              <div
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => handleNavigation(item)}
              >
                <IconComponent size={18} />
                <span>{item.label}</span>
              </div>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

export default Header;